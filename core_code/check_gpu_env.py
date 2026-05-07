import json

from runtime_config import detect_runtime_info


def main():
    info = detect_runtime_info()
    dump = dict(info)
    dump["device"] = str(dump["device"])
    print("ASS GPU Environment Check")
    print(json.dumps(dump, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
