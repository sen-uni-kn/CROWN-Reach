import subprocess
import os
import sys
import signal
from crownreach.attack import attack


def check_ready_signal(server_process):
    while True:
        output = server_process.stdout.readline()
        if output == "" and server_process.poll() is not None:
            returncode = server_process.returncode
            error_output = (
                server_process.stderr.read()
                if server_process.stderr
                else "No stderr available"
            )
            raise RuntimeError(
                f"Server process exited unexpectedly with return code {returncode}. Error output:\n{error_output}"
            )
        if "Server started." in output:
            print("Server Started")
            return True


def run_crownreach(config_path, test_mode=False, here_dir="."):
    cmd = [f"{here_dir}/CrownSettings", config_path]
    if test_mode:
        cmd.append("--test")
    status = None
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    ) as proc:
        try:
            # Real-time output display
            while True:
                output = proc.stdout.readline()
                if output == "":
                    if proc.poll() is not None:
                        break

                output = output.strip()
                as_status_line = output.lower().replace(".", "")
                if as_status_line == "verified":
                    status = "Verified"
                elif as_status_line == "unsafe":
                    status = "Unsafe"
                elif as_status_line == "unknown":
                    status = "Unknown"

                if output:
                    print(output)  # Print each line of stdout
            # Once the loop exits, check if there were any errors
            stderr_output = proc.stderr.read()
            if stderr_output:
                print("Errors:\n", stderr_output.strip())
                status = "Error"
        except Exception as e:
            print("An error occurred:", e)
            proc.kill()
            proc.wait()
            status = "Error"
    if status is None:
        status = "Unknown"
    return status


def kill_process_on_port(port):
    try:
        # Find the PID of the process using the port
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"])
        pids = result.decode().strip().split("\n")
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process {pid} on port {port}")
    except subprocess.CalledProcessError:
        pass


def verify(config_path: str, port: int = 5000, test_mode: bool = False):
    # ------------------- run attack -------------------- #
    attack_result, trajectory = attack(config_path)
    if attack_result:
        print("Falsified.")
        return False, trajectory
    else:
        print("Attack results unknown, start verifying.")

        # ------------------- check and kill process on port -------------------- #
        kill_process_on_port(port)

        # ------------------- start server -------------------- #
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        here_dir = os.path.dirname(os.path.realpath(__file__))
        server_process = subprocess.Popen(
            ["python3", f"{here_dir}/crown.py", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # make sure server is running
        check_ready_signal(server_process)

        # ------------------- run verification -------------------- #
        status = run_crownreach(config_path, test_mode=test_mode, here_dir=here_dir)

        # ------------------- close server -------------------- #
        try:
            # Try to terminate the server gracefully
            server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill the server if it doesn't terminate gracefully
            os.kill(server_process.pid, signal.SIGKILL)
            server_process.wait()

        print("Server Closed")
        print("=" * 100)
        print(f"Verification Result: {status}")
        return status


if __name__ == "__main__":
    # ------------------- config_path -------------------- #
    config_path = sys.argv[1]
    port = 5000  # specify your server port

    # ------------------- check if test mode -------------------- #
    test_mode = "--test" in sys.argv

    verify(config_path, port, test_mode)
