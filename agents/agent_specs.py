AGENTS = [
    {
        "name":"Literature Searcher",
        "json_schema": """{
            "query": "string",
            "is_need_online_search": "boolean",
            "papers": [{"title":"string","authors":"string","year":"string","doi":"string","url":"string"}],
            "notes": "string"
        }"""
    },
    {
        "name":"ChemProcess Modeler",
        "role_default": (
        "You are ChemProcess Modeler. Construct mathematical models for process design. "
        "After answering, explicitly state whether new variables are defined or existing variables are modified "
        "for equations/mass balances. In the final JSON, set `variables_changed` to true or false accordingly."
    ),
        "json_schema": """{
            "models": [{"name":"string","formula":"string","assumptions":"string"}],
            "variables_changed": "boolean",
            "variables_used": "boolean",
            "notes": "string"
        }"""
    },
    {
        "name":"Experiment Designer",
        "json_schema": """{
            "designs": [{"factors":"string","levels":"string","method":"string"}],
            "notes": "string"
        }"""
    },
    {
        "name":"Fitting Wizard",
        "json_schema": """{
            "fitting_method":"string",
            "parameters":{"param":"value"},
            "fit_quality":"string",
            "notes":"string"
        }"""
    },
    {
        "name":"Optimization Navigator",
        "json_schema": """{
            "objectives":["string"],
            "constraints":["string"],
            "optimal_conditions":"string",
            "notes":"string"
        }"""
    },
    {
    "name": "Process Analyzer",
    "desc": "Analyze process flowsheets, perform material/energy balances, identify bottlenecks, integration opportunities and KPIs, and propose actionable improvements.",
    "json_schema": """{
        "stage": "string",
        "flowsheet_summary": "string",
        "balances": { "material": "string", "energy": "string" },
        "bottlenecks": [ "string" ],
        "opportunities": [ "string" ],
        "kpis": { "Li2CO3_purity": "number", "Li_yield": "number", "specific_energy": "number" },
        "recommendations": [ "string" ]
    }"""
    }
]
