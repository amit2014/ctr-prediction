# add headers to train data
train_headers='1iclick\tweekday\thour\ttimestamp\tlog_type\tuser_id\tuser_agent\tip\tregion\tcity\tad_exchange\tdomain\turl\tanonymous_url_id\tad_slot_id\tad_slot_width\tad_slot_height\tad_slot_visibility\tad_slot_format\tad_slot_floor_price\tcreative_id\tkey_page_url\tadvertiser_id\tuser_tags'
sed -i $train_headers train_data.txt

# add headers to test data
test_headers='1iweekday\thour\ttimestamp\tlog_type\tuser_id\tuser_agent\tip\tregion\tcity\tad_exchange\tdomain\turl\tanonymous_url_id\tad_slot_id\tad_slot_width\tad_slot_height\tad_slot_visibility\tad_slot_format\tad_slot_floor_price\tcreative_id\tkey_page_url\tadvertiser_id\tuser_tags'
sed -i $test_headers test_data.txt
