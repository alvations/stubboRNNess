import sys
import argparse

def getWrongLabels(pred):
	wrongs = []
	for i in range(0, len(pred)):
		if pred[i]!='0' and pred[i]!='1':
			wrongs.append(i)
	return wrongs

if __name__=='__main__':

	#Parse arguments:
	description = 'Validation script for Task 11: Complex Word Identification.'
	description += ' The dataset file must be in the format provided by the task organizers.'
	description += ' The predicted labels file must contain one label 0 or 1 per line, and have the same number of lines as the dataset.'
	epilog = 'Returns: SUCCESS if the predicted labels file is in the correct format, FAILURE otherwise.'
	parser=argparse.ArgumentParser(description=description, epilog=epilog)
	parser.add_argument('--gold', required=True, help='File containing a dataset.')
	parser.add_argument('--pred', required=True, help='File containing predicted labels.')
	args = vars(parser.parse_args())

	#Retrieve labels:
	gold = [line.strip() for line in open(args['gold'])]
	pred = [line.strip() for line in open(args['pred'])]

	#Count wrong labels:
	wrongs = getWrongLabels(pred)

	#Print result:
	if len(wrongs)==0 and len(pred)==len(gold):
		print('SUCCESS: Label file is in the correct format!')
	else:
		print('FAILURE: Label file is not in the correct format.\nErrors:')
		if len(wrongs)>0:
			for wrong in wrongs:
				print('\tUnrecognized label in line ' + str(wrong) + ': ' + str(pred[wrong]) + ' (should be 0 or 1).')
		if len(pred)!=len(gold):
			print('\tFile contains ' + str(len(pred)) + ' lines instead of ' + str(len(gold)) + '.')
