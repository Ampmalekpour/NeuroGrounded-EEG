# deep4net_hgd_baseline_full22.py
#
# Full 22-channel Deep4Net baseline on HGD: the "22ch" row of the 2-class and
# 4-class tables, plus per-subject ERD/ERS reference maps (pre-cue baseline).
#
# Because every (seed, subject, fold) is seeded independently, the saliency script's
# own 22-channel training reproduces these numbers (up to GPU non-determinism), so
# there is a single canonical 22ch row.

import os

import deep4net_hgd_common as C


def run(class_mode):
    records, _ = C.run_experiment("baseline_full22", class_mode, C.full_montages())
    out_dir = C.experiment_dir("baseline_full22", class_mode)

    topo = C.make_topo_info(C.FULL_CHANNELS)
    for sid in C.SUBJECT_IDS:
        erd = C.get_erd_maps(sid, C.FULL_CHANNELS)
        for k, v in erd.items():
            C.save_array(os.path.join(out_dir, f"{k}_S{sid:02d}.npy"), v)
        C.plot_erd_ers_topomaps(erd["ERD_L"], erd["ERD_R"], erd["ERD_(L-R)"], erd["ERD_comb"],
                                topo, os.path.join(out_dir, f"erd_S{sid:02d}.png"))

    C.write_text_report(os.path.join(out_dir, "baseline_full22_report.txt"),
                        f"Deep4Net - HGD - FULL 22-CHANNEL BASELINE - {class_mode}", records)
    return C.summarize(records)


if __name__ == "__main__":
    results = {cm: run(cm) for cm in C.CLASS_MODES}
    print("\n" + "=" * 80 + "\nSUMMARY - FULL 22-CHANNEL BASELINE\n" + "=" * 80)
    for cm, s in results.items():
        print(f"{cm}: {s['grand_mean']*100:.2f}% +/- {s['grand_std_across_subjects']*100:.2f}% "
              f"(chance {s['chance_acc']*100:.1f}%, majority {s['majority_class_acc_mean']*100:.1f}%)")
