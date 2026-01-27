from typing import Literal

VENDOR_DICT = {
    'ALLY AUTO'               : {'vendor': 'Ally Auto'          , 'category': 'Automotive'   },
    'CITY OF DALLAS UTILITIES': {'vendor': 'City of Dallas'     , 'category': 'Utilities'    },
    'GOOGLE STORAGE'          : {'vendor': 'Google'             , 'category': 'Subscriptions'},
    'GREYSTAR'                : {'vendor': 'Greystar'           , 'category': 'Rent'         },
    'NETFLIX'                 : {'vendor': 'Netflix'            , 'category': 'Subscriptions'},
    'PLANET FITNESS'          : {'vendor': 'Planet Fitness'     , 'category': 'Gym & Fitness'},
    'PLNT FITNESS'            : {'vendor': 'Planet Fitness'     , 'category': 'Gym & Fitness'},
    'SPOTIFY'                 : {'vendor': 'Spotify'            , 'category': 'Subscriptions'},
    'TXU ENERGY'              : {'vendor': 'TXU Energy'         , 'category': 'Utilities'    },
    'WSJ DIGITAL'             : {'vendor': 'Wall Street Journal', 'category': 'Subscriptions'}
}

ExpenseCategory = Literal[
    'Automotive',
    'Baby & Child',
    'Books',
    'Clothing & Accessories',
    'Electronics & Accessories',
    'General',
    'Groceries',
    'Gym & Fitness',
    'Health & Personal Care',
    'Home',
    'Office Supplies',
    'Other Food & Beverage',
    'Pet Supplies',
    'Sports & Recreation',
    'Other'
]