import sys
import argparse

def evaluateIdentifier(gold, pred):
	"""
	Performs an intrinsic evaluation of a Complex Word Identification approach.

	@param gold: A vector containing gold-standard labels.
	@param pred: A vector containing predicted labels.
	@return: Precision, Recall and F-1.
	"""
	
	#Initialize variables:
	precisionc = 0
	precisiont = 0
	recallc = 0
	recallt = 0
	
	#Calculate measures:
	for i in range(0, len(gold)):
		gold_label = gold[i]
		predicted_label = pred[i]
		if gold_label==predicted_label:
			precisionc += 1
			if gold_label==1:
				recallc += 1
		if gold_label==1:
			recallt += 1
		precisiont += 1
	
	precision = float(precisionc)/float(precisiont)
	recall = float(recallc)/float(recallt)
	fmean = 0.0
	if precision==0.0 and recall==0.0:
		fmean = 0.0
	else:
		fmean = 2*(precision*recall)/(precision+recall)
		
	#Return measures:
	return precision, recall, fmean

if __name__=='__main__':

	#Parse arguments:
	description = 'Evaluation script for Task 11: Complex Word Identification.'
	description += ' The gold-standard file is a dataset with labels in the format provided by the task organizers.'
	description += ' The predicted labels file must contain one label 0 or 1 per line, and must have the same number of lines as the gold-standard.'
	epilog = 'Returns: Precision, Recall and F1.'
	parser=argparse.ArgumentParser(description=description, epilog=epilog)
	parser.add_argument('--gold', required=True, help='File containing dataset with gold-standard labels.')
	parser.add_argument('--pred', required=True, help='File containing predicted labels.')
	args = vars(parser.parse_args())

	#Retrieve labels:
	gold = [int(line.strip().split('\t')[3]) for line in open(args['gold'])]
	pred = [int(line.strip()) for line in open(args['pred'])]

	#Calculate scores:
	p, r, f = evaluateIdentifier(gold, pred)

	#Present scores:
	print('Precision: ' + str(p))
	print('Recall: ' + str(r))
	print('F1: ' + str(f))

