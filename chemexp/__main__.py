import sys
from .server import server


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("Usage: chemexp <path>\n",
              "<path> must contain chemprop models/checkpoints (.pt files)",
              file=sys.stderr)
        exit(1)
    else:
        server.set_models_dir(sys.argv[1])
        server.run(debug=True)
