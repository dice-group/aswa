import pandas as pd
import argparse
import os
pd.set_option('display.max_columns', 20)
pd.set_option("display.precision", 2)

parser = argparse.ArgumentParser(description='Analysis')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--model', type=str, default='WideResNet28x10')
parser.add_argument('--ratio', type=str, default='005')
args = parser.parse_args()

exp_name = f"Original_{args.dataset}_{args.model}_1000E_Val_{args.ratio}_ratio"

a = f"{args.dataset}{args.model}/FirstRun/"
b = f"{args.dataset}{args.model}/SecondRun/"
c = f"{args.dataset}{args.model}/ThirdRun/"

data_frames=[]

if os.path.exists(f"{a}{exp_name}/results.csv"):
    data_frames.append(pd.read_csv(f"{a}{exp_name}/results.csv", index_col=0))
    print(data_frames[0])
else:
    print(f"{a}{exp_name}/results.csv does not exist.")


if os.path.exists(f"{b}{exp_name}/results.csv"):
    data_frames.append(pd.read_csv(f"{b}{exp_name}/results.csv", index_col=0))
    print(data_frames[-1])
else:
    print(f"{b}{exp_name}/results.csv does not exist.")


if os.path.exists(f"{c}{exp_name}/results.csv"):
    data_frames.append(pd.read_csv(f"{c}{exp_name}/results.csv", index_col=0))
    print(data_frames[-1])
else:
    print(f"{c}{exp_name}/results.csv does not exist.")

dfs = pd.concat(data_frames).groupby("ep", as_index=False)

print(dfs)

def selected_epochs(x):
    return x[(x["ep"] == 50) | (x["ep"] == 100) | (x["ep"] == 200) | (x["ep"] == 300) | (
            x["ep"] == 400)
             | (x["ep"] == 500)
             | (x["ep"] == 600)
             | (x["ep"] == 700)
             | (x["ep"] == 800)
             | (x["ep"] == 900)
             | (x["ep"] == 1000)][["ep", "test_acc", "swa_test_acc", "aswa_test_acc"]]


def show_as_reported(x):
    # Base
    print(" & ".join([f"{i:.2f}" for i in x["test_acc"].tolist()]))
    # SWA
    print(" & ".join([f"{i:.2f}" for i in x["swa_test_acc"].tolist()]))
    # ASWA
    print(" & ".join([f"{i:.2f}" for i in x["aswa_test_acc"].tolist()]))


def show_two_together(x, y):
    assert len(x) == len(y)
    # Base
    print(f"{args.model} & " + " & ".join(
        [f"${i:.1f} \pm {ii:.1f}$" for i, ii in zip(x["test_acc"].tolist(), y["test_acc"].tolist())]))
    # SWA
    print(f"{args.model}-SWA & " + " & ".join(
        [f"${i:.1f} \pm {ii:.1f}$" for i, ii in zip(x["swa_test_acc"].tolist(), y["swa_test_acc"].tolist())]))
    # ASWA
    print(f"{args.model}-ASWA & " + " & ".join(
        [f"${i:.1f} \pm {ii:.1f}$" for i, ii in zip(x["aswa_test_acc"].tolist(), y["aswa_test_acc"].tolist())]))


#dfs = pd.concat([df_a, df_b, df_c]).groupby("ep", as_index=False)


show_two_together(selected_epochs(dfs.mean()), selected_epochs(dfs.std()))
