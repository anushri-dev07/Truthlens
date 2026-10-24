from truthlens.data.io import probe_video


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to video file")
    args = p.parse_args()
    info = probe_video(args.video)
    print(info)


if __name__ == "__main__":
    main()
