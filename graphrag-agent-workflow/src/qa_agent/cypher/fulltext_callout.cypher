CALL db.index.fulltext.queryNodes('callout_fulltext', $query) YIELD node, score
WITH node AS c, score
OPTIONAL MATCH (s:Section)-[:HAS_CALLOUT]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  c.id            AS node_id,
  'Callout'       AS node_label,
  c.indexed_text  AS indexed_text,
  c.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
