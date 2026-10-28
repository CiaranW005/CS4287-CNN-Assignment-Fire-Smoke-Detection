
def print_evaluation(history):
    m = history.evaluation

    print("\n" + "=" * 72)
    print(" " * 24 + "MODEL EVALUATION")
    print("=" * 72)

    print(f"\nOverall mAP50:     {m.map50:.4f}")
    print(f"Overall mAP50-95:  {m.map50_95:.4f}")

    print("\nDetection performance by object size")
    print("-" * 72)
    print(f"{'Size':<12} {'mAP':>12} {'mAR':>12} {'Recall@50':>14} {'Recall@75':>14}")
    print("-" * 72)

    rows = [
        ("Overall", m.map50_95, m.mar_100, m.recall50, m.recall75),
        ("Small", m.map_small, m.mar_small, m.recall50_small, m.recall75_small),
        ("Medium", m.map_medium, m.mar_medium, m.recall50_medium, m.recall75_medium),
        ("Large", m.map_large, m.mar_large, m.recall50_large, m.recall75_large),
    ]

    for name, ap, ar, r50, r75 in rows:
        print(f"{name:<12} {ap:>12.4f} {ar:>12.4f} {r50:>14.4f} {r75:>14.4f}")

    print("\nPer-class mAP50-95")
    print("-" * 72)

    for name, value in zip(("Smoke", "Fire"), m.map_per_class):
        print(f"{name:<12} {value:.4f}")

    print("=" * 72)