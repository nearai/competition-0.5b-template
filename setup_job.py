import shutil
import threading
import time
from pathlib import Path
from subprocess import PIPE, run

import nearai
import nearai.log
import nearai.registry

TARGET = "llm-c-124M-train"

api = nearai.log.LogCLI()


def log(message: str, **kwargs):
    print(message)
    api.push(TARGET, message=message, **kwargs)


def prepare_command(np=8, **kwargs):
    command = ["mpirun", "-np", str(np), "./train_gpt2cu"]
    for key, value in kwargs.items():
        command.append(f"-{key}")
        command.append(value)
    return command


SHUTDOWN = False


def check_logs():
    last_line = -1
    while not SHUTDOWN:
        time.sleep(3)
        with open("log124M/main.log", "r") as f:
            for i, line in enumerate(f):
                if i > last_line:
                    log(line, source="log124M/main.log", line_number=str(i))
                    last_line = i


def main():
    log("Downloading starter pack")

    try:
        starter_pack = nearai.registry.registry.download(
            "au.near/llm-c-starter-pack/0.0.2"
        )
        fineweb = nearai.registry.registry.download("au.near/fineweb-sample-1B/0.0.1")
    except Exception as e:
        log("Error", error=str(e))
        exit(1)

    log("Starter pack downloaded", path=str(starter_pack))

    llm_c = Path(__file__).parent.resolve()

    shutil.copy2(starter_pack / "gpt2_tokenizer.bin", llm_c)
    shutil.copy2(starter_pack / "gpt2_124M_bf16.bin", llm_c)

    log("Started pack copied")

    command = prepare_command(
        np=8,
        i=f"{fineweb}/fineweb_train_*.bin",
        j=f"{fineweb}/fineweb_val_*.bin",
        o="log124M",
        e="d12",
        b="64",
        t="1024",
        d="524288",
        r="1",
        z="1",
        c="0.1",
        l="0.0006",
        q="0.0",
        u="700",
        n="5000",
        v="250",
        s="20000",
        h="1",
    )
    log("Running llm.c on 8 GPUs", command=" ".join(command))

    threading.Thread(target=check_logs).start()

    try:
        proc = run(
            command,
            cwd=llm_c,
            stdout=PIPE,
            stderr=PIPE,
        )
    except Exception as e:
        log("Error running llm.c", error=str(e))
        exit(1)

    global SHUTDOWN
    SHUTDOWN = True

    log(
        "Finish running llm.c",
        returncode=proc.returncode,
        stdout=proc.stdout.decode(),
        stderr=proc.stderr.decode(),
    )


if __name__ == "__main__":
    main()
