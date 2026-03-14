import os
import shutil

def merge_two_dirs(src_a, src_b, dst, tag_a="A", tag_b="B"):
    os.makedirs(dst, exist_ok=True)

    def copy_with_rename(src_dir, tag):
        copied = 0
        conflict = 0
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".npy"):
                continue
            sp = os.path.join(src_dir, fn)
            dp = os.path.join(dst, fn)

            if not os.path.isfile(sp):
                continue

            if os.path.exists(dp):
                # 同名冲突：改名保存
                base, ext = os.path.splitext(fn)
                new_name = f"{base}__from_{tag}{ext}"
                dp = os.path.join(dst, new_name)
                conflict += 1

            shutil.copy2(sp, dp)
            copied += 1
        return copied, conflict

    c1, k1 = copy_with_rename(src_a, tag_a)
    c2, k2 = copy_with_rename(src_b, tag_b)

    print(f"\nMerged into: {dst}")
    print(f"  From {src_a}: copied={c1}, renamed_due_to_conflict={k1}")
    print(f"  From {src_b}: copied={c2}, renamed_due_to_conflict={k2}")
    print(f"  Total copied: {c1 + c2}, total conflicts handled: {k1 + k2}")

if __name__ == "__main__":
    # root = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/train"
    # root = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/norm/valid"

    root = "/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/train"

    # # covid: ori + 1b
    # merge_two_dirs(
    #     src_a=os.path.join(root, "covid"),
    #     src_b=os.path.join(root, "covid1b"),
    #     dst=os.path.join(root, "covid_ori_and_1b"),
    #     tag_a="covid",
    #     tag_b="covid1b",
    # )

    # # non-covid: ori + 1b
    # merge_two_dirs(
    #     src_a=os.path.join(root, "non-covid"),
    #     src_b=os.path.join(root, "non-covid1b"),
    #     dst=os.path.join(root, "non-covid_ori_and_1b"),
    #     tag_a="non-covid",
    #     tag_b="non-covid1b",
    # )

    # covid: ori + 1b
    merge_two_dirs(
        src_a=os.path.join(root, "covid"),
        src_b=os.path.join(root, "covid1b"),
        dst=os.path.join(root, "covid_ori_and_1b"),
        tag_a="covid",
        tag_b="covid1b",
    )

    # non-covid: ori + 1b
    merge_two_dirs(
        src_a=os.path.join(root, "non-covid"),
        src_b=os.path.join(root, "non-covid1b"),
        dst=os.path.join(root, "non-covid_ori_and_1b"),
        tag_a="non-covid",
        tag_b="non-covid1b",
    )