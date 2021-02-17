#read input file
fin = open("in.txt", "rt")
#read file contents to string
data = fin.read()
#replace all occurrences of the required string
data = data.replace('tensor([', '')
data = data.replace("], device='cuda:0')", ';')
data = data.replace(',', ';')
data = data.replace(' ', '')

#close the input file
fin.close()
#open the input file in write mode
fout = open("out.txt", "wt")
#overrite the input file with the resulting data
fout.write(data)
#close the file
fin.close()