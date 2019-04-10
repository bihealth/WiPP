WiPP: create

create:
	@echo 'creating the 'WiPP' conda environment'
	@echo
	conda env create --file environment.yml
	@echo

clean :
	conda remove --name WiPP --all 
	find ./projects/example_project/data/ -type f ! -name 'README.md' -exec rm -f {} +