def read_data_from_file(filename):
    data = []
    sent_counter = 0
    with open(filename) as infile:
        lines = infile.readlines()
        sent_t = ([], [])
        for line in lines:
            if line == '\n':
                data.append(sent_t)
                sent_t = ([], [])
                sent_counter += 1
            else:
                word, pos_tag, chunk_tag = line.strip().split()
                sent_t[0].append(word)
                sent_t[1].append(chunk_tag)
    return data

