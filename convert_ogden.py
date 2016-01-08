from __future__ import print_function

import io, sys

with io.open(sys.argv[1], 'r', encoding='latin-1') as fin:
	for line in fin:
		line = line.encode('utf8', 'replace').strip()
		if line != '/':
			print(line.split('/')[0])
		else:
			print(line)
