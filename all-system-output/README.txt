
This folder includes the system output of our models (i.e. EncDecA, EncDecA-RF, EncDecA-RF-LS) 
and other models in comparision (i.e. PBMT-R, Hybrid, SBMT-SARI).

We also include the complex sentences (Complex) and their references (Reference).

1. System output on Newsela dataset
Newsela
	test        -- output on the whole test set
		Complex
		Dress
		Dress-Ls
		EncDecA
		Hybrid
		PBMT-R
		Reference
	eval        -- output of the 100 randomly chosen complex sentences (on test set) used for human evaluation
		...
		...
		...

2. System output on WikiSmall dataset
WikiSmall
	test        -- output on the whole test set
		...
		...
		...
	eval        -- output of the 100 randomly chosen complex sentences (on test set) used for human evaluation
		...
		...
		...

3. System output on WikiLarge dataset
There are totally 8 references for each complex sentence. We ramdomly choose one as Reference.
WikiLarge
	test        -- output on the whole test set
		...
		...
	eval        -- output of the 100 randomly chosen complex sentences (on test set) used for human evaluation
		...
		...




