//! Release-only supervisor: `liquid-agent supervise` keeps the installed server running.
//!
//! It is started, detached, by `liquid-agent open`, so only installed (release) copies are supervised; `cargo run`
//! and `npm run dev` start the server directly. It restarts the server after a crash, with a growing delay, and
//! gives up when the server keeps crashing. A stop that the server meant (`shutdown()` leaves
//! `server.clean-exit` in the data folder) is not a crash: the supervisor waits for an update handover to start the
//! next build, follows that server instead, and exits when none comes back.
use anyhow::Result;
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

/// Shared with liquid-server's main.rs.
pub const CLEAN_EXIT_FILE: &str = "server.clean-exit";
pub const SERVER_PID_FILE: &str = "server.pid";
pub const SUPERVISOR_PID_FILE: &str = "supervisor.pid";

/// Delays before each restart within the crash window.
const BACKOFF: [u64; 5] = [1, 2, 5, 10, 30];
/// More crashes than this within the window, and the supervisor stops trying.
const MAX_CRASHES: usize = 5;
const CRASH_WINDOW: Duration = Duration::from_secs(10 * 60);
/// How long to wait for a new server after a clean stop (an update handover takes up to about 80 s).
const DEFAULT_GRACE: Duration = Duration::from_secs(120);

pub struct Supervisor {
    pub server: PathBuf,
    pub app: PathBuf,
    pub data: PathBuf,
    pub port: u16,
    pub log: PathBuf,
}

/// True while the process exists. EPERM means it exists under another user.
pub fn alive(pid: i32) -> bool {
    pid > 1
        && unsafe {
            libc::kill(pid, 0) == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
        }
}

pub fn read_pid(path: &Path) -> Option<i32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// On unless `LIQUID_AGENT_SUPERVISE=0` (or false/no/off) is set when `liquid-agent open` runs.
pub fn enabled() -> bool {
    !turned_off(std::env::var("LIQUID_AGENT_SUPERVISE").ok().as_deref())
}

fn turned_off(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Stops a running supervisor so it does not restart the server that is about to be stopped. Returns whether one
/// was running.
pub fn stop(data: &Path) -> bool {
    let file = data.join(SUPERVISOR_PID_FILE);
    let Some(pid) = read_pid(&file).filter(|pid| alive(*pid)) else {
        let _ = fs::remove_file(&file);
        return false;
    };
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    for _ in 0..50 {
        if !alive(pid) {
            break;
        }
        sleep(Duration::from_millis(100));
    }
    let _ = fs::remove_file(&file);
    true
}

enum Watched {
    Child(Child),
    Adopted(i32),
}

impl Supervisor {
    fn note(&self, line: &str) {
        if let Ok(mut log) = OpenOptions::new().create(true).append(true).open(&self.log) {
            let _ = writeln!(log, "supervisor: {line}");
        }
    }

    fn spawn(&self, open: bool) -> Result<Child> {
        // A marker from an earlier stop must not make a crash during start-up look clean.
        let _ = fs::remove_file(self.data.join(CLEAN_EXIT_FILE));
        let output = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log)?;
        let mut command = Command::new(&self.server);
        if open {
            command.arg("--open");
        }
        Ok(command
            .env("LIQUID_AGENT_PORT", self.port.to_string())
            .env("LIQUID_AGENT_HOME", &self.data)
            .env("LIQUID_AGENT_ROOT", &self.app)
            .env("LIQUID_AGENT_SUPERVISED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(output.try_clone()?))
            .stderr(Stdio::from(output))
            .spawn()?)
    }

    /// The server to watch: the one already answering on the port (after an update, or started before this
    /// supervisor), or a new one.
    fn watch(&self, open: bool) -> Result<Watched> {
        if crate::launcher::running(self.port) {
            if let Some(pid) = read_pid(&self.data.join(SERVER_PID_FILE)).filter(|p| alive(*p)) {
                return Ok(Watched::Adopted(pid));
            }
        }
        Ok(Watched::Child(self.spawn(open)?))
    }

    /// Waits for a server to come back after a clean stop. True when one is answering.
    fn wait_for_handover(&self, grace: Duration) -> bool {
        let until = Instant::now() + grace;
        while Instant::now() < until {
            if crate::launcher::running(self.port) {
                return true;
            }
            sleep(Duration::from_millis(500));
        }
        false
    }

    pub fn run(&self) -> Result<()> {
        let me = std::process::id() as i32;
        let pid_file = self.data.join(SUPERVISOR_PID_FILE);
        fs::create_dir_all(&self.data)?;
        if read_pid(&pid_file).is_some_and(|pid| pid != me && alive(pid)) {
            return Ok(()); // Another supervisor already looks after this install.
        }
        fs::write(&pid_file, me.to_string())?;
        let grace = std::env::var("LIQUID_AGENT_SUPERVISOR_GRACE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_GRACE);
        let result = self.supervise(grace, &pid_file, me);
        if read_pid(&pid_file) == Some(me) {
            let _ = fs::remove_file(&pid_file);
        }
        result
    }

    fn supervise(&self, grace: Duration, pid_file: &Path, me: i32) -> Result<()> {
        let mut crashes: VecDeque<Instant> = VecDeque::new();
        let mut open = true;
        loop {
            let watched = self.watch(open)?;
            open = false; // Only the first start opens the browser.
            let status: Option<ExitStatus> = match watched {
                Watched::Child(mut child) => Some(child.wait()?),
                Watched::Adopted(pid) => {
                    while alive(pid) {
                        sleep(Duration::from_secs(1));
                    }
                    None
                }
            };
            // A newer supervisor took over (or `stop` removed the file): leave the server to it.
            if read_pid(pid_file) != Some(me) {
                return Ok(());
            }
            if self.data.join(CLEAN_EXIT_FILE).is_file() {
                self.note(
                    "the server stopped cleanly; waiting for an update to start the next build",
                );
                if self.wait_for_handover(grace) {
                    continue;
                }
                self.note("no server came back; the supervisor stops");
                return Ok(());
            }
            let how = status.map_or("its process ended".to_owned(), describe);
            let Some(delay) = restart_delay(&mut crashes, Instant::now()) else {
                self.note(&format!(
                    "the server stopped unexpectedly ({how}) {} times in 10 minutes; not restarting it. Run `liquid-agent logs`, then `liquid-agent open`.",
                    crashes.len()
                ));
                return Ok(());
            };
            self.note(&format!(
                "the server stopped unexpectedly ({how}); restarting it in {delay} s"
            ));
            sleep(Duration::from_secs(delay));
        }
    }
}

/// Records a crash at `now` and returns the seconds to wait before restarting, or None to give up. Crashes older
/// than the window are forgotten, so a server that has been stable for a while gets the short delay again.
fn restart_delay(crashes: &mut VecDeque<Instant>, now: Instant) -> Option<u64> {
    crashes.push_back(now);
    while crashes
        .front()
        .is_some_and(|at| now.duration_since(*at) > CRASH_WINDOW)
    {
        crashes.pop_front();
    }
    (crashes.len() <= MAX_CRASHES).then(|| BACKOFF[(crashes.len() - 1).min(BACKOFF.len() - 1)])
}

fn describe(status: ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("exit code {code}"),
        (None, Some(signal)) => format!("signal {signal}"),
        _ => "unknown status".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restarts_back_off_then_give_up_and_recover_after_the_window() {
        let start = Instant::now();
        let mut crashes = VecDeque::new();
        let delays: Vec<_> = (0..6)
            .map(|i| restart_delay(&mut crashes, start + Duration::from_secs(i)))
            .collect();
        assert_eq!(
            delays,
            [Some(1), Some(2), Some(5), Some(10), Some(30), None]
        );
        // Eleven minutes later the old crashes no longer count.
        let later = start + Duration::from_secs(11 * 60);
        assert_eq!(restart_delay(&mut crashes, later), Some(1));
        assert_eq!(crashes.len(), 1);
    }

    #[test]
    fn supervision_is_on_unless_turned_off() {
        assert!(!turned_off(None));
        assert!(!turned_off(Some("1")));
        assert!(!turned_off(Some("")));
        assert!(turned_off(Some("0")));
        assert!(turned_off(Some(" OFF ")));
        assert!(turned_off(Some("false")));
    }
}
