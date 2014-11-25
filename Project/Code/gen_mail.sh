j=1
for i in fwd/*
do
	python eml.py $i $j
	((j++))
done