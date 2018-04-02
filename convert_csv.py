import csv
with open("raw.txt", "r") as input, open("format.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["No.", "P1", "P2", "P3", "P4", "Duration"])
    num_sentence = 0
    for line1 in input:
        line2 = next(input)
        line1 = line1.replace("\n","")
        line2 = line2.replace("\n","")
        num_phoneme = 0
        phoneme_list = ["sil", "sil", "sil", "sil"]
        for elem1, elem2 in zip(line1.split(" "), line2.split(" ")):
            phoneme_list.pop(0)
            phoneme_list.append(elem1)
            #  ''.join(str(item) + "," for item in phoneme_list).strip('\"')
            writer.writerow([str(num_sentence) + "_" + str(num_phoneme), phoneme_list[0], phoneme_list[1], phoneme_list[2], phoneme_list[3], elem2])
            num_phoneme += 1
        
        num_sentence += 1
