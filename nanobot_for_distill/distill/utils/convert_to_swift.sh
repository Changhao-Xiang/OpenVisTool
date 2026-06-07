
python distill/utils/convert_to_swift.py \
  --sessions-dir workspaces/qwen35plus/VinciCoder \
  --index-file workspaces/qwen35plus/VinciCoder-filtered-index-12k.jsonl \
  --output dataset/OpenVisTool/Web2html-all-12k.jsonl \
  --tools 'read_file,write_file,edit_file,list_dir,exec,crop,locate_in_crop,draw_bbox,draw_line,draw_circle,in_range_color,rotate,flip,enhance_contrast,adjust_brightness,detect_edges,grayscale,connected_components,find_contours,hough_lines,hough_circles,template_match,render_html' \
  --workers 1