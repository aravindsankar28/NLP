for i in {0..9}
do
    python build_index.py big.txt.0$i spellchecker.model.0$i > op$i &
done
