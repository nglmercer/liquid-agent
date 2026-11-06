//! Integration tests for [`InferenceManager`] driven by the fixture engine.

use std::{
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use futures_util::StreamExt;
use liquid_inference::{
    fixture::{FixtureConfig, FixtureCounters, FixtureEngine},
    ChatMessage, ChatRequest, DevicePreference, EventStream, InferenceError, InferenceEvent,
    InferenceManager, LoadSpec, ModelRole, ModelState,
};
use tokio_util::sync::CancellationToken;

fn spec(role: ModelRole, max_concurrent: usize) -> LoadSpec {
    LoadSpec {
        role,
        weights: PathBuf::from("/fixture/model.gguf"),
        projector: None,
        context_tokens: 512,
        max_concurrent,
        device: DevicePreference::Auto,
    }
}

fn req() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::text("user", "hi")], 16)
}

fn manager_with(config: FixtureConfig) -> (InferenceManager, Arc<FixtureCounters>) {
    let engine = Arc::new(FixtureEngine::new(config));
    let counters = engine.counters();
    (InferenceManager::new(engine), counters)
}

fn expect_err(result: Result<EventStream, InferenceError>) -> InferenceError {
    match result {
        Ok(_) => panic!("expected generate to fail"),
        Err(err) => err,
    }
}

fn load_count(counters: &FixtureCounters) -> usize {
    counters.loads.load(Ordering::SeqCst)
}

#[tokio::test]
async fn load_reports_ready_and_capabilities() {
    let config = FixtureConfig::default();
    let expected_caps = config.capabilities;
    let (m, counters) = manager_with(config);

    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("load should succeed");

    assert_eq!(m.status(ModelRole::Small), ModelState::Ready);
    assert_eq!(m.capabilities(ModelRole::Small), Ok(expected_caps));
    assert_eq!(load_count(&counters), 1);
    assert_eq!(m.status(ModelRole::Omni), ModelState::Unloaded);
}

#[tokio::test]
async fn generate_yields_text_usage_and_finished_in_order() {
    let (m, _counters) = manager_with(FixtureConfig::default());
    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("load should succeed");

    let stream = m
        .generate(ModelRole::Small, req(), CancellationToken::new())
        .expect("generate should start");
    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("all events should be Ok");

    assert_eq!(events.len(), 4, "events: {events:?}");
    assert_eq!(events[0], InferenceEvent::TextDelta("Hello".to_string()));
    assert_eq!(events[1], InferenceEvent::TextDelta(" world".to_string()));
    assert!(
        matches!(events[2], InferenceEvent::Usage { .. }),
        "expected Usage, got {:?}",
        events[2]
    );
    assert_eq!(events[3], InferenceEvent::Finished);
}

#[tokio::test]
async fn second_load_while_ready_is_invalid_and_reload_after_unload_works() {
    let (m, counters) = manager_with(FixtureConfig::default());
    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("first load should succeed");

    let err = m
        .load(spec(ModelRole::Small, 1))
        .await
        .expect_err("second load should fail");
    assert_eq!(
        err,
        InferenceError::InvalidTransition {
            role: ModelRole::Small,
            from: "Ready".to_string(),
            to: "Loading".to_string(),
        }
    );
    assert_eq!(m.status(ModelRole::Small), ModelState::Ready);

    m.unload(ModelRole::Small)
        .await
        .expect("unload should succeed");
    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("load after unload should succeed");

    assert_eq!(m.status(ModelRole::Small), ModelState::Ready);
    assert_eq!(load_count(&counters), 2);
}

#[tokio::test]
async fn generate_on_unloaded_role_is_not_loaded() {
    let (m, _counters) = manager_with(FixtureConfig::default());

    let err = expect_err(m.generate(ModelRole::Omni, req(), CancellationToken::new()));
    assert_eq!(err, InferenceError::NotLoaded(ModelRole::Omni));
}

#[tokio::test]
async fn cancel_yields_cancelled_then_ends_and_status_recovers() {
    let config = FixtureConfig {
        token_delay: Duration::from_millis(50),
        tokens: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ],
        ..Default::default()
    };
    let (m, _counters) = manager_with(config);
    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("load should succeed");

    let cancel = CancellationToken::new();
    let mut stream = m
        .generate(ModelRole::Small, req(), cancel.clone())
        .expect("generate should start");

    let first = stream.next().await;
    assert_eq!(first, Some(Ok(InferenceEvent::TextDelta("a".to_string()))));

    cancel.cancel();

    let second = stream.next().await;
    assert_eq!(second, Some(Err(InferenceError::Cancelled)));
    assert!(
        stream.next().await.is_none(),
        "stream should end after Cancelled"
    );

    assert_eq!(m.status(ModelRole::Small), ModelState::Ready);

    let second_stream = m
        .generate(ModelRole::Small, req(), CancellationToken::new())
        .expect("second generate should succeed after cancel");
    let events: Vec<_> = second_stream.collect::<Vec<_>>().await;
    assert!(events.iter().all(|e| e.is_ok()), "events: {events:?}");
    assert_eq!(
        events.last(),
        Some(&Ok(InferenceEvent::Finished)),
        "second stream should finish normally"
    );
}

#[tokio::test]
async fn single_slot_is_busy_while_stream_alive() {
    let (m, _counters) = manager_with(FixtureConfig::default());
    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("load should succeed");

    let first = m
        .generate(ModelRole::Small, req(), CancellationToken::new())
        .expect("first generate should start");

    let err = expect_err(m.generate(ModelRole::Small, req(), CancellationToken::new()));
    assert_eq!(err, InferenceError::Busy(ModelRole::Small));
    assert_eq!(m.status(ModelRole::Small), ModelState::Busy);

    drop(first);

    let _third = m
        .generate(ModelRole::Small, req(), CancellationToken::new())
        .expect("generate should succeed after the first stream is dropped");
}

#[tokio::test]
async fn unload_waits_for_in_flight_generation_and_refuses_new_work() {
    let config = FixtureConfig {
        token_delay: Duration::from_millis(20),
        ..Default::default()
    };
    let (m, counters) = manager_with(config);
    m.load(spec(ModelRole::Small, 1))
        .await
        .expect("load should succeed");

    let stream = m
        .generate(ModelRole::Small, req(), CancellationToken::new())
        .expect("generate should start");

    let unloader = m.clone();
    let unload_task = tokio::spawn(async move { unloader.unload(ModelRole::Small).await });

    let mut saw_unloading = false;
    for _ in 0..200 {
        if m.status(ModelRole::Small) == ModelState::Unloading {
            saw_unloading = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(saw_unloading, "status never became Unloading");

    let err = expect_err(m.generate(ModelRole::Small, req(), CancellationToken::new()));
    assert_eq!(err, InferenceError::Busy(ModelRole::Small));

    let events: Vec<_> = stream.collect::<Vec<_>>().await;
    assert!(events.iter().all(|e| e.is_ok()), "events: {events:?}");

    unload_task
        .await
        .expect("unload task should not panic")
        .expect("unload should succeed once drained");

    assert_eq!(m.status(ModelRole::Small), ModelState::Unloaded);
    assert_eq!(counters.models_dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn engine_load_error_is_reported_and_status_is_failed() {
    let config = FixtureConfig {
        load_error: Some("boom".to_string()),
        ..Default::default()
    };
    let (m, _counters) = manager_with(config);

    let err = m
        .load(spec(ModelRole::Small, 1))
        .await
        .expect_err("load should fail");
    match err {
        InferenceError::Load { message, .. } => assert_eq!(message, "boom"),
        other => panic!("expected Load error, got {other:?}"),
    }
    assert_eq!(
        m.status(ModelRole::Small),
        ModelState::Failed("boom".to_string())
    );
}

#[tokio::test]
async fn unsupported_model_message_is_preserved() {
    let config = FixtureConfig {
        unsupported: Some("q2_k not supported".to_string()),
        ..Default::default()
    };
    let (m, _counters) = manager_with(config);

    let err = m
        .load(spec(ModelRole::Small, 1))
        .await
        .expect_err("load should fail");
    assert_eq!(
        err,
        InferenceError::Unsupported("q2_k not supported".to_string())
    );
    assert!(matches!(m.status(ModelRole::Small), ModelState::Failed(_)));
}

#[tokio::test]
async fn zero_max_concurrent_is_rejected_before_state_changes() {
    let (m, counters) = manager_with(FixtureConfig::default());

    let err = m
        .load(spec(ModelRole::Small, 0))
        .await
        .expect_err("max_concurrent 0 should be rejected");
    assert_eq!(
        err,
        InferenceError::Unsupported("max_concurrent must be at least 1".to_string())
    );
    assert_eq!(m.status(ModelRole::Small), ModelState::Unloaded);
    assert_eq!(load_count(&counters), 0);
}

#[tokio::test]
async fn concurrent_loads_allow_at_most_one_success() {
    let config = FixtureConfig {
        load_delay: Duration::from_millis(100),
        ..Default::default()
    };
    let (m, counters) = manager_with(config);

    let first = m.clone();
    let second = m.clone();
    let t1 = tokio::spawn(async move { first.load(spec(ModelRole::Small, 1)).await });
    let t2 = tokio::spawn(async move { second.load(spec(ModelRole::Small, 1)).await });

    let r1 = t1.await.expect("first load task should not panic");
    let r2 = t2.await.expect("second load task should not panic");

    let results = [r1, r2];
    let oks = results.iter().filter(|r| r.is_ok()).count();
    assert!(oks <= 1, "at most one load may succeed, got {oks}");
    for result in &results {
        if let Err(err) = result {
            assert!(
                matches!(err, InferenceError::InvalidTransition { .. }),
                "unexpected error: {err:?}"
            );
        }
    }

    assert_eq!(m.status(ModelRole::Small), ModelState::Ready);
    assert_eq!(load_count(&counters), 1);
}
