Each instance in the training dataset is distributed in the following format:

<sentence> <word> <index> <label>

All components are separated by a tabulation marker.
The <sentence> component is a sentence extracted from a certain source.
The <word> component is a word in <sentence>.
The <index> component is the position of <word> in the tokens of <sentence>.
The <label> component is a label that receives value 1 if the word is complex, and value 0 otherwise.