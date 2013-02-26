int eat_callback(int (*callback)(int, int)) {
    return callback(5, 2);
}