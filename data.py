def collect_infos_from_instance(instance):
    clients = []
    demands = []
    path = "instances/" + instance + ".txt"
    with open(path) as f:
        line = f.readline()
        row = line.split(' ')
        dimension = int(row[-1])
        line = f.readline()
        row = line.split(' ')
        capacity = int(row[-1])
        indices = []
        f.readline()
        for i in range(dimension):
            line = f.readline()
            if line == '':
                break
            row = line.split(' ')
            indices.append(int(row[0]))
            clients.append([float(row[1]), float(row[2])])
        f.readline()
        for i in range(dimension):
            line = f.readline()
            if line == '':
                break
            row = line.split(' ')
            demands.append(int(row[1]))
    f.close()
    return dimension, capacity, indices, clients, demands