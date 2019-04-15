WiPP: create testdata

create:
	@echo 'creating the 'WiPP' conda environment'
	@echo
	conda env create --file environment.yml
	@echo

testdata:
	@echo 'fetching test data:'
	wget https://file-public.bihealth.org/transient/wipp/WiPP_files.tar.gz
	@echo 'unpacking data:'
	tar -xvzf WiPP_files.tar.gz
	@echo 'copying training samples:'
	cp WiPP_files/sample1.cdf ./projects/example_project/data/training_samples
	cp WiPP_files/sample2.cdf ./projects/example_project/data/training_samples
	@echo 'copying optimization samples:'
	cp WiPP_files/sample3.cdf ./projects/example_project/data/optimization_samples
	cp WiPP_files/sample4.cdf ./projects/example_project/data/optimization_samples
	@echo 'moving samples:'
	mv -t ./projects/example_project/data/Condition1 WiPP_files/sample1.cdf WiPP_files/sample3.cdf WiPP_files/sample5.cdf
	mv -t ./projects/example_project/data/Condition2 WiPP_files/sample2.cdf WiPP_files/sample4.cdf WiPP_files/sample6.cdf
	@echo 'moving classifiers:'
	mv -t ./projects/example_project/example_classifier/ WiPP_files/XCMS-*
	@echo 'removing empty folder:'
	rm -r WiPP_files
clean :
	conda remove --name WiPP --all 
	find ./projects/example_project/data/ -type f ! -name 'README.md' -exec rm -f {} +
