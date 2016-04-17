
def prepend_line(filename, header):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(header.rstrip('\r\n') + '\n' + content)


if __name__ == '__main__':
    header = 'weekday\thour\ttimestamp\tlog_type\tuser_id\tuser_agent\tip\tregion\tcity\tad_exchange\tdomain\turl' \
        + '\tanonymous_url_id\tad_slot_id\tad_slot_width\tad_slot_height\tad_slot_visibility\tad_slot_format' \
        + '\tad_slot_floor_price\tcreative_id\tkey_page_url\tadvertiser_id\tuser_tags'
    prepend_line('train_data.txt', 'click\t' + header)
    prepend_line('test_data.txt', header)