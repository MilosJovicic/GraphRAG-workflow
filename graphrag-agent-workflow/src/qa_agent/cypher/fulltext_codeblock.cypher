CALL db.index.fulltext.queryNodes('codeblock_fulltext', $query) YIELD node, score
WITH node AS b, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_CODE]->(b)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($language IS NULL OR b.language = $language)
  AND ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  b.id            AS node_id,
  'CodeBlock'     AS node_label,
  b.indexed_text  AS indexed_text,
  b.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  COALESCE('[' + b.language + '] ' + s.breadcrumb, s.breadcrumb) AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
