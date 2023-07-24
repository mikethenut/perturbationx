import os
import csv


def preprocess_datasets(folder_path, header_labels=('logFC', 't', 'adj.P.Val')):
    positive_samples = []
    negative_samples = []

    for file in os.listdir(folder_path):
        if file.endswith(".tsv"):
            with open(os.path.join(folder_path, file), "r") as in_file:
                reader = csv.reader(in_file, delimiter="\t")
                header = next(reader)

                log_fc_idx = header.index(header_labels[0])
                t_idx = header.index(header_labels[1])

                for row in reader:
                    t = float(row[t_idx])
                    log_fc = float(row[log_fc_idx])

                    if log_fc > 0:
                        positive_samples.append((log_fc, t))
                    elif log_fc < 0:
                        negative_samples.append((log_fc, t))

    with open(folder_path + "/positive_samples.csv", "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["log_fc", "t"])
        writer.writerows(positive_samples)

    with open(folder_path + "/negative_samples.csv", "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["log_fc", "t"])
        writer.writerows(negative_samples)


if __name__ == "__main__":
    preprocess_datasets("../../data/ExpressionExamples")
    preprocess_datasets("../../data/COPD1", header_labels=('foldChange', 't', None))
