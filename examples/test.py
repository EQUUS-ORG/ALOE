import aloe

engine = aloe.aloe(input_file = "test.csv")
engine.add_step(aloe.StereoIsoConfig()) # Generate stereoisomers
engine.add_step(aloe.ConformerConfig()) # Ember conformers
engine.add_step(aloe.OptConfig()) # Optimize conformers
engine.add_step(aloe.RankConfig(k=3)) # Rank optimized conformers, pick the best 3
engine.add_step(aloe.ThermoConfig()) # Thermochemistry calculations via ASE
output_file = engine.run() # Asynchronous execution

print(output_file)