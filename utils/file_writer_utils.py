def write_token_id_arrays_to_text_file(token_ids_array, filename, tokenizer):
    """
    Writes an array of token id arrays as decoded text into a file.

    :param token_ids_array: The array with the token id arrays.
    :param filename: The name of the file.
    :param tokenizer: The tokenizer to decode the token ids.
    :return: None
    """
    with open(filename, 'w') as file:
        for token_ids in token_ids_array:
            text = tokenizer.map_id_to_text(token_ids)
            file.write("%s\n" % text)
