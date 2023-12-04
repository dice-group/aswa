import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option("display.precision", 2)

model_name = "VGG16"
ratio="02"
exp_name = f"Original_CIFAR10_{model_name}_1000E_Val_{ratio}_ratio"

a = "CIFAR10VGG/FirstRun/"
b = "CIFAR10VGG/SecondRun/"
c = "CIFAR10VGG/ThirdRun/"

df_a = pd.read_csv(f"{a}{exp_name}/results.csv", index_col=0)
df_b = pd.read_csv(f"{b}{exp_name}/results.csv", index_col=0)
df_c = pd.read_csv(f"{c}{exp_name}/results.csv", index_col=0)


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
    print(f"{model_name} & " + " & ".join(
        [f"${i:.1f} \pm {ii:.1f}$" for i, ii in zip(x["test_acc"].tolist(), y["test_acc"].tolist())]))
    # SWA
    print(f"{model_name}-SWA & " + " & ".join(
        [f"${i:.1f} \pm {ii:.1f}$" for i, ii in zip(x["swa_test_acc"].tolist(), y["swa_test_acc"].tolist())]))
    # ASWA
    print(f"{model_name}-ASWA & " + " & ".join(
        [f"${i:.1f} \pm {ii:.1f}$" for i, ii in zip(x["aswa_test_acc"].tolist(), y["aswa_test_acc"].tolist())]))


dfs = pd.concat([df_a, df_b, df_c]).groupby("ep", as_index=False)

show_two_together(selected_epochs(dfs.mean()), selected_epochs(dfs.std()))
