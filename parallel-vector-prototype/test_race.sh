for i in {0..100}
do
    if ! python test_logit.py
    then
        exit
    fi
done
