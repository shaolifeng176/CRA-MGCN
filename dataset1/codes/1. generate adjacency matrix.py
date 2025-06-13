import csv
import os

def read_herb_pairs_from_csv(file_path):
    herb_pairs = []
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    # Suppose that the format of the drug pair is ' herb 1 herb 2'
                    if len(row) == 1:
                        parts = row[0].split(' ')
                        if len(parts) == 2:
                            herb_pairs.append((parts[0].strip(), parts[1].strip()))
            if herb_pairs:
                print(f"Successfully read the drug pair: {herb_pairs}")
            break
        except UnicodeDecodeError:
            print(f"Encoding {encoding} cannot be used to read files.")
            continue
    return herb_pairs

# Save the adjacency matrix file
def save_adjacency_matrix(data, file_path):
    pairs = [f"{a} {b}" for a, b in data]
    pairs_set = [frozenset(pair.split()) for pair in pairs]
    edges = set()
    for i, pair_i in enumerate(pairs_set):
        for j, pair_j in enumerate(pairs_set):
            if i != j and not pair_i.isdisjoint(pair_j):
                edges.add((pairs[i], pairs[j]))

    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        for edge in edges:
            writer.writerow(edge)

input_csv_file = 'D:\\Project\\pythonProject\\re\\keti\dataset1\\dataset\\herb pairs for training\\all herb pairs for training.txt'

herb_pairs = read_herb_pairs_from_csv(input_csv_file)

output_dir = 'D:\\Project\\pythonProject\\re\\keti\\dataset1\\dataset\\training data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_adjacency_matrix(herb_pairs, os.path.join(output_dir, 'HCGCN-all herb pairs1.cites'))

print(f"Adjacency matrix saved to {os.path.join(output_dir, 'HCGCN-all herb pairs.cites')}")
