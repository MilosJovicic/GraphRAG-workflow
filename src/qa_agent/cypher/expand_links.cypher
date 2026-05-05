UNWIND $seeds AS seed
MATCH (s:Section {id: seed.node_id})-[:LINKS_TO|LINKS_TO_PAGE]->(t)
WHERE t.id IS NOT NULL OR t.url IS NOT NULL
WITH seed.node_id AS seed_id, t,
     CASE WHEN 'Section' IN labels(t) THEN 'Section' ELSE 'Page' END AS tlabel
RETURN
  COALESCE(t.id, t.url)         AS node_id,
  tlabel                        AS node_label,
  COALESCE(t.indexed_text, t.title, t.path) AS indexed_text,
  COALESCE(t.text, t.description, t.title)  AS raw_text,
  CASE WHEN tlabel = 'Page' THEN t.url ELSE null END AS url,
  t.anchor                      AS anchor,
  t.breadcrumb                  AS breadcrumb,
  COALESCE(t.title, t.breadcrumb) AS title,
  seed_id                       AS seed_id
LIMIT $cap
