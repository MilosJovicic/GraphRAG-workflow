UNWIND $seeds AS seed
OPTIONAL MATCH (s:Section)-[r:DEFINES|MENTIONS]->(e)
  WHERE e.name = seed.node_id OR e.key = seed.node_id
OPTIONAL MATCH (src:Section {id: seed.node_id})-[r2:DEFINES|MENTIONS]->(e2)
WITH seed.node_id AS seed_id,
     collect(DISTINCT s) + collect(DISTINCT e2) AS hits
UNWIND hits AS hit
WITH seed_id, hit
WHERE hit IS NOT NULL
WITH seed_id, hit, head(labels(hit)) AS hlabel
RETURN
  COALESCE(hit.id, hit.name, hit.key, hit.url) AS node_id,
  hlabel                  AS node_label,
  COALESCE(hit.indexed_text, hit.text, hit.description, hit.name) AS indexed_text,
  COALESCE(hit.text, hit.description, hit.name) AS raw_text,
  null                    AS url,
  hit.anchor              AS anchor,
  hit.breadcrumb          AS breadcrumb,
  COALESCE(hit.title, hit.name, hit.breadcrumb) AS title,
  seed_id                 AS seed_id
LIMIT $cap
