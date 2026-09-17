import argparse

from .server import build


def main():
    parser = argparse.ArgumentParser(description="Ganglion MCP stdio bridge")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--client-id")
    args = parser.parse_args()
    build(args.endpoint, observer=args.observer, client_id=args.client_id).run()


if __name__ == "__main__":
    main()
