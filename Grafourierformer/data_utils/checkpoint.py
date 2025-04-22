import csv
import os


def results_to_file(args, val, test, best_test):

    if not os.path.exists('./results_ablation_pe/{}'.format(args.dataset)):
        print("=" * 20)
        print("Create Results File !!!")

        os.makedirs('./results_ablation_pe/{}'.format(args.dataset))

    filename = "./results_ablation_pe/{}/result.csv".format(
        args.dataset)

    headerList = ["Method", "Layer-Num", "Slope", "n_hop", "gamma", "drop_out", "attn_drop", "drop_path",
                  "::::::::", "val", "test", "best_test"]

    with open(filename, "a+") as f:

        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, {}, {}, {}, :::::::::, {:.5f}, {:.5f}, {:.5f}\n".format(
            args.model_type, args.num_layers, args.slope, args.n_hop, args.gamma, args.dropout,
            args.attn_dropout, args.drop_prob, val, test, best_test)
        f.write(line)