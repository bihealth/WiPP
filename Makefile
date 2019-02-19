WiPP: create testdata

create:
	@echo 'creating the 'WiPP' conda environment'
	@echo
	conda env create --file environment.yml
	@echo

testdata:
	@echo 'fetching test data:'
	@echo 'cirrhosis samples:'
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_02.CDF -P ./projects/example_project/data/Cirrhosis
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_03.CDF -P ./projects/example_project/data/Cirrhosis
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_04.CDF -P ./projects/example_project/data/Cirrhosis
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_06.CDF -P ./projects/example_project/data/Cirrhosis
	@echo 'HCC samples:'
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_01.CDF -P ./projects/example_project/data/HCC
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_05.CDF -P ./projects/example_project/data/HCC
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_07.CDF -P ./projects/example_project/data/HCC
	wget ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS105/GC-SIM-MS_09.CDF -P ./projects/example_project/data/HCC
	@echo 'copying training samples:'
	cp ./projects/example_project/data/HCC/GC-SIM-MS_01.CDF ./projects/example_project/data/training_samples
	cp ./projects/example_project/data/Cirrhosis/GC-SIM-MS_02.CDF ./projects/example_project/data/training_samples
	@echo 'copying optimization samples:'
	cp ./projects/example_project/data/Cirrhosis/GC-SIM-MS_03.CDF ./projects/example_project/data/optimization_samples
	cp ./projects/example_project/data/HCC/GC-SIM-MS_05.CDF ./projects/example_project/data/optimization_samples

clean :
	conda remove --name WiPP --all 
	find ./projects/example_project/data/ -type f ! -name 'README.md' -exec rm -f {} +