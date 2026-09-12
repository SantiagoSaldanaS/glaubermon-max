"""Small development-only OU pool, separate from frozen evaluation teams.

These are hand-selected legal practice sets, not a tournament-strength claim.
The engine validates every team before a rollout. Species overlap with evaluation
is unavoidable in OU; these team compositions and several sets differ. Original
checkpoint training data may already contain evaluation sets (provenance unknown).
"""
TRAINING_TEAMS = {
    'development_offense': ']'.join([
        'Dragonite||heavydutyboots|multiscale|dragondance,extremespeed,earthquake,icepunch|Adamant|,252,4,,,252|||||,,,,,Normal',
        'Meowscarada||choicescarf|protean|flowertrick,knockoff,uturn,tripleaxel|Jolly|,252,4,,,252|||||,,,,,Grass',
        'Zamazenta||leftovers|dauntlessshield|irondefense,bodypress,crunch,substitute|Jolly|252,,4,,,252|||||,,,,,Steel',
        'Heatran||airballoon|flashfire|magmastorm,earthpower,taunt,stealthrock|Timid|,,,252,4,252|||||,,,,,Grass',
        'Iron Valiant||lifeorb|quarkdrive|moonblast,aurasphere,shadowball,calmmind|Timid|,,,252,4,252|||||,,,,,Fairy',
        'Great Tusk||leftovers|protosynthesis|earthquake,knockoff,rapidspin,bulkup|Jolly|252,,4,,,252|||||,,,,,Steel',
    ]),
    'development_rain': ']'.join([
        'Pelipper||damprock|drizzle|surf,hurricane,uturn,roost|Bold|248,,252,,8,|||||,,,,,Ground',
        'Barraskewda||choiceband|swiftswim|liquidation,flipturn,closecombat,aquajet|Adamant|,252,4,,,252|||||,,,,,Water',
        'Raging Bolt||boosterenergy|protosynthesis|thunderclap,dracometeor,thunderbolt,calmmind|Modest|64,,,252,,192|||||,,,,,Fairy',
        'Iron Treads||leftovers|quarkdrive|earthquake,voltswitch,rapidspin,stealthrock|Jolly|252,,4,,,252|||||,,,,,Water',
        'Gholdengo||leftovers|goodasgold|shadowball,makeitrain,recover,nastyplot|Modest|252,,,252,4,|||||,,,,,Water',
        'Kingambit||lumberry|supremeoverlord|kowtowcleave,suckerpunch,ironhead,swordsdance|Adamant|252,252,,,,4|||||,,,,,Dark',
    ]),
}
