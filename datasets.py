datasets = {
    "issa": {
        "type": "rdf",
        "url": "https://data-issa.cirad.fr/sparql",
        "agrovoc": """
                prefix oa: <http://www.w3.org/ns/oa#>
                prefix skosxl: <http://www.w3.org/2008/05/skos-xl#> 
                SELECT distinct ?article (?uri as ?label) 
                from <http://agrovoc.fao.org/graph> 
                from <http://data-issa.cirad.fr/graph/thematic-descriptors> 
                from <http://data-issa.cirad.fr/graph/annif-descriptors>
                WHERE { 
                    ?s oa:hasTarget ?article ; oa:hasBody ?uri . 
                } limit 10000 offset %s
            """,
        "wikidata": ""
    },
    "covid": {
        "type": "rdf",
        "url": "http://covidontheweb.inria.fr/sparql",
        "wikidata": """
            prefix wdt:     <http://www.wikidata.org/prop/direct/>
            prefix schema:  <http://schema.org/>
            prefix oa:      <http://www.w3.org/ns/oa#> 
            prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> 
            prefix dct:     <http://purl.org/dc/terms/> 
            SELECT distinct ?article ?label 
            FROM <http://ns.inria.fr/covid19/graph/entityfishing> 
            FROM named <http://ns.inria.fr/covid19/graph/wikidata-named-entities-full>
            FROM <http://ns.inria.fr/covid19/graph/articles>
            WHERE { 
                ?x schema:about ?article; oa:hasBody ?body. 
                
                GRAPH <http://ns.inria.fr/covid19/graph/wikidata-named-entities-full>{ 
                    ?body rdfs:label ?label. 
                } 
                
                ?article dct:issued ?date; dct:abstract [ rdf:value ?abs ].     
            } 
            LIMIT 10000 OFFSET  %s
        """
    },
    "crobora": {
        "type": "api",
        "url": "http://dataviz.i3s.unice.fr/crobora-api",
        "labels": ["http://dataviz.i3s.unice.fr/crobora-api/cluster/names", "http://dataviz.i3s.unice.fr/crobora-api/cluster/names2"],
        "images": "http://dataviz.i3s.unice.fr/crobora-api/search/imagesOR?%s&options=illustration&options=location&options=celebrity&options=event" 

    }
}
