import sys
import fileinput

FILENAME=sys.argv[1]

for line in fileinput.input(inplace=True, backup='.bak', files=FILENAME):
	line_edited = line.replace(chr(9601)+"< mas k >", "<mask>")
	print(line_edited, end='')
