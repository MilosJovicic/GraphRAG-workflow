CALL db.index.fulltext.queryNodes('tablerow_fulltext', $query) YIELD node, score
WITH node AS r, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE_ROW]->(r)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  r.id            AS node_id,
  'TableRow'      AS node_label,
  r.indexed_text  AS indexed_text,
  r.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
