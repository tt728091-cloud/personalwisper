// Native entry point for PersonalWisper.app. It starts the Python app as a child process so macOS
// attributes Microphone / Accessibility / Input Monitoring permissions to PersonalWisper.app.
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef PROJECT_DIR
#error "build with -DPROJECT_DIR=\"/path/to/personalwisper\""
#endif

extern char **environ;
static pid_t child;

static void forward_signal(int sig) {
    if (child > 0) kill(child, sig);
}

int main(void) {
    // Tell the Python app where its bundle is, for Restart and Start at Login.
    char exe[PATH_MAX], app[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) == 0 && realpath(exe, app)) {
        for (int i = 0; i < 3; i++) {  // strip Contents/MacOS/PersonalWisper
            char *slash = strrchr(app, '/');
            if (slash) *slash = '\0';
        }
        setenv("PERSONALWISPER_APP", app, 1);
    }
    // Apps launched from Finder get a minimal PATH; ffmpeg (for very long recordings) lives in Homebrew.
    setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    if (chdir(PROJECT_DIR) != 0) return 1;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, PROJECT_DIR "/logs/personalwisper.log",
                                     O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

    char *argv[] = {PROJECT_DIR "/.venv/bin/python", PROJECT_DIR "/personalwisper.py", NULL};
    if (posix_spawn(&child, argv[0], &actions, NULL, argv, environ) != 0) {
        perror("posix_spawn");
        return 1;
    }
    signal(SIGTERM, forward_signal);
    signal(SIGINT, forward_signal);
    signal(SIGHUP, forward_signal);

    int status;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
