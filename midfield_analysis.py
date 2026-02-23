import numpy as np

# ----------------------------
# 1️⃣ Data Setup
# ----------------------------

np.random.seed(42)

leagues = np.array([
    "Primeira Liga",
    "La Liga",
    "EPL",
    "Bundesliga",
    "Ligue 1",
    "Serie A"
])

metrics = np.array([
    "Tackles", "Interceptions", "Prog Carries", "Prog Passes",
    "Key Passes", "Assists", "Non-Pen Goals", "Pass %"
])

players_size = 10

primeiraliga_midfielders_name = ["Alan Viera", "Hjulmand", "Sudakov", "Luis Esteves", "Jean Gorby", "Gustavo Sa", "Beni", "Vasco Souza", "Amorim", "Brandon Aguilera"]
laliga_midfielders_name = ["Pedri", "Jude Bellingham", "Pablo Barrios", "Pablo Fornals", "Santi Cornesana", "Jauregizar", "Edu Exposito", "Iliax Moriba", "Lucien Agoume", "Luis Milla"]
epl_midfielders_name = ["Enzo Fernandes", "Bruno Fernandes", "Declan Rice", "Tijani Reijnders", "Dominik Szoboszlai", "Xabi Simons", "Bruno Guimaraes", "Adam Wharton", "Elliot Anderson", "James Garner"]
bundesliga_midfielders_name = ["Kimmich", "Nicolas Seiwald", "Felix Nmecha", "Aleix Garcia", "Wonter Burger", "Angelo Stiller", "Can Uzun", "Rani Khedira", "Johan Manzambi", "Isak Johannesson"]
ligueone_midfielders_name = ["Thomasson", "Vitinha", "Hojbjerg", "Haraldsson", "Corentin Tolisso", "Valentin Rongier", "Valentin Barco", "Cristian Casseres", "Lammine Camara", "Himad Abdelli"]
serieA_midfielders_name = ["Hakan Calhanoghu", "Luka Modric", "Andre Zambo", "Manu Kone", "Khephren Thuram", "Remo Freuler", "Nico Paz", "Matteo Guendouzi", "Ismael Kone", "Aurthur Atta"]

midfielders_name = np.array((
    primeiraliga_midfielders_name,
    laliga_midfielders_name,
    epl_midfielders_name,
    bundesliga_midfielders_name,
    ligueone_midfielders_name,
    serieA_midfielders_name,
))

# Random realistic stats
data = np.zeros((6, 10, 8))

# Defensive stats
data[:,:,0] = np.random.randint(20, 120, (6,10))   # Tackles
data[:,:,1] = np.random.randint(10, 80, (6,10))    # Interceptions

# Ball progression
data[:,:,2] = np.random.randint(30, 200, (6,10))   # Prog Carries
data[:,:,3] = np.random.randint(50, 300, (6,10))   # Prog Passes

# Creative stats
data[:,:,4] = np.random.randint(10, 100, (6,10))   # Key Passes
data[:,:,5] = np.random.randint(0, 15, (6,10))     # Assists
data[:,:,6] = np.random.randint(0, 12, (6,10))     # Goals

# Passing %
data[:,:,7] = np.random.randint(70, 95, (6,10))

# ----------------------------
# 2️⃣ Normalization
# ----------------------------

def normalize_stats(raw_data):
    """Min-Max normalize across all leagues & players per metric"""
    min_per_metric = np.min(raw_data, axis=(0,1))
    max_per_metric = np.max(raw_data, axis=(0,1))
    diff = max_per_metric - min_per_metric
    normalized = (raw_data - min_per_metric[np.newaxis, np.newaxis, :]) / diff[np.newaxis, np.newaxis, :]
    return normalized

normalized_data = normalize_stats(data)

# ----------------------------
# 3️⃣ League & Overall Averages
# ----------------------------

def league_average(league_index):
    avg = np.mean(normalized_data[league_index, :, :])
    return avg

for idx, league in enumerate(leagues):
    print(f"{league} midfielders average performance: {np.round(league_average(idx), 3)}")

# Overall average per league for ranking
overall_avg = np.mean(normalized_data, axis=(1,2))
ranking = np.argsort(-overall_avg)

print("\nRank of leagues by overall performance:")
for n, idx in enumerate(ranking):
    print(f"  {n+1}. {leagues[idx]} - {np.round(overall_avg[idx], 3)}")

# ----------------------------
# 4️⃣ Player vs League Comparison
# ----------------------------

def player_vs_league(player_index, league_index, metric_index):
    diff = normalized_data[league_index, player_index, metric_index] - np.mean(normalized_data[league_index,:,metric_index])
    direction = "more" if diff > 0 else "less"
    return diff, direction

# Example: 10th La Liga midfielder progressive carries
diff_val, direction = player_vs_league(9, 1, 2)
print(f"\n{midfielders_name[1,9]} has {np.abs(np.round(diff_val,2))} {direction} {metrics[2]} than the La Liga average")

# ----------------------------
# 5️⃣ Best Midfielders by Role & Overall
# ----------------------------

def best_midfielders(stats_slice, role_name):
    """Rank top 5 players based on stats slice (normalized)"""
    stats_sum = np.sum(stats_slice, axis=2)
    flat_ranking = np.argsort(-stats_sum.ravel())
    flat_names = np.ravel(midfielders_name)
    top_five = flat_names[flat_ranking[:5]]
    print(f"\nTop 5 {role_name} midfielders:")
    for i, name in enumerate(top_five):
        print(f"  {i+1}. {name}")
    return top_five

# Attacking
best_midfielders(normalized_data[:, :, 4:7], "Attacking")

# Defensive
best_midfielders(normalized_data[:, :, 0:2], "Defensive")

# Creative
best_midfielders(normalized_data[:, :, 3:6], "Creative")

# Overall
best_midfielders(normalized_data[:, :, :], "Overall")

# ----------------------------
# 6️⃣ League-Level Role Analysis
# ----------------------------

def league_role_index(stats_slice):
    """Returns the index of the league with highest average role performance"""
    role_score = np.mean(stats_slice, axis=(1,2))
    return np.argmax(role_score)

creative_league = league_role_index(normalized_data[:, :, 3:6])
defensive_league = league_role_index(normalized_data[:, :, 0:2])
attacking_league = league_role_index(normalized_data[:, :, 4:7])

print(f"\n{leagues[creative_league]} produces the most creative midfielders")
print(f"{leagues[defensive_league]} produces the most defensive midfielders")
print(f"{leagues[attacking_league]} produces the most attacking midfielders")

# ----------------------------
# 7️⃣ Balanced Players & Leagues
# ----------------------------

# Player balance: low std across metrics = balanced
player_balance = np.std(normalized_data, axis=2)
most_balanced_player_index = np.argmin(player_balance.mean(axis=1))
print(f"\n{np.ravel(midfielders_name)[most_balanced_player_index]} is the most balanced player")

# League balance: low avg std = balanced league
league_balance = player_balance.mean(axis=1)
most_balanced_league_index = np.argmin(league_balance)
print(f"{leagues[most_balanced_league_index]} is the most balanced league")

# League with more specialized players = high std
most_specialized_league_index = np.argmax(league_balance)
print(f"{leagues[most_specialized_league_index]} produces more specialized midfielders")


# ----------------------------
# 8️⃣ REPORT SUMMARY SECTION
# ----------------------------

print("\n" + "="*60)
print("        FOOTBALL MIDFIELD ANALYTICS REPORT")
print("="*60)

# --- League Overall Ranking ---
print("\n📊 LEAGUE OVERALL PERFORMANCE RANKING")
print("-"*60)

for rank, idx in enumerate(ranking):
    print(f"{rank+1:>2}. {leagues[idx]:<15} | Score: {overall_avg[idx]:.3f}")

# --- League Role Strengths ---
print("\n🎯 LEAGUE ROLE SPECIALIZATION")
print("-"*60)
print(f"Most Creative League   : {leagues[creative_league]}")
print(f"Most Defensive League  : {leagues[defensive_league]}")
print(f"Most Attacking League  : {leagues[attacking_league]}")
print(f"Most Balanced League   : {leagues[most_balanced_league_index]}")
print(f"Most Specialized League: {leagues[most_specialized_league_index]}")

# --- Top Overall Midfielders ---
print("\n⭐ TOP 5 OVERALL MIDFIELDERS")
print("-"*60)

overall_stats_sum = np.sum(normalized_data, axis=2)
overall_rank = np.argsort(-overall_stats_sum.ravel())
flat_names = np.ravel(midfielders_name)

for i in range(5):
    print(f"{i+1:>2}. {flat_names[overall_rank[i]]}")

# --- Top Creative Players ---
print("\n🎨 TOP 5 CREATIVE MIDFIELDERS")
print("-"*60)

creative_stats_sum = np.sum(normalized_data[:, :, 3:6], axis=2)
creative_rank = np.argsort(-creative_stats_sum.ravel())

for i in range(5):
    print(f"{i+1:>2}. {flat_names[creative_rank[i]]}")

# --- Top Defensive Players ---
print("\n🛡 TOP 5 DEFENSIVE MIDFIELDERS")
print("-"*60)

defensive_stats_sum = np.sum(normalized_data[:, :, 0:2], axis=2)
defensive_rank = np.argsort(-defensive_stats_sum.ravel())

for i in range(5):
    print(f"{i+1:>2}. {flat_names[defensive_rank[i]]}")

# --- Most Balanced Player ---
print("\n⚖ MOST BALANCED MIDFIELDER")
print("-"*60)
print(f"{flat_names[most_balanced_player_index]}")

print("\n" + "="*60)
print("              END OF REPORT")
print("="*60)