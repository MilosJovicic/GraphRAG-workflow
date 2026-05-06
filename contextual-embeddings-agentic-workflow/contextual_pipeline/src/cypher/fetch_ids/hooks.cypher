MATCH (n:Hook)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source'    THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.name AS node_id
