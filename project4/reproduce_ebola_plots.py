"""
Topic 2 Task 0: Reproduce Ebola Plots
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_ebola_data(filename):
    filepath = os.path.join('data', filename)
    try :
        with open(filepath, 'r') as f:
            lines = f.readlines()

        dates = []
        new_cases = []

        for line in lines[1:]:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    dates.append(parts[0])
                    try:
                        new_cases.append(float(parts[2]))
                    except ValueError:
                        new_cases.append(0.0)
        first_date = datetime.strptime(dates[0], "%Y-%m-%d")
        days = []
        for date_str in dates:
            current_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_since = (current_date - first_date).days
            days.append(days_since)

        days = np.array(days)
        new_cases = np.array(new_cases)
        progress = np.cumsum(new_cases)

        return days, new_cases, progress
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
def plot_ebola_data(country, days, new_cases, progress):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # plot new cases
    ax1.scatter(days, new_cases, 
               color='red', s=80, facecolors='none', 
               edgecolors='red', linewidth=2, alpha=0.7,
              label='Number of outbreaks', zorder=5)
    ax1.set_xlabel('Days since first outbreak', fontsize=12)
    ax1.set_ylabel('Number of outbreaks', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xlim([0, max(days) + 50])
    ax1.set_ylim([0, max(new_cases) * 1.15])
    ax1.grid(True, alpha=0.3)

    #plot the progress
    ax2 = ax1.twinx()
    ax2.plot(days, progress,
            marker='s', markersize=4, linewidth=1.5,
            color='black', markerfacecolor='black',
            markeredgecolor='black', markeredgewidth=1,
            label='Cumulative number of outbreaks', zorder=4)
    ax2.set_ylabel('Cumulative number of outbreaks', 
                  fontsize=12, color='black', rotation=270, labelpad=20)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([0, max(progress) * 1.15])
    
    plt.title(f'Ebola outbreaks in {country}', fontsize=14, fontweight='bold')

    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'ebola_plot_{country.lower().replace(" ", "_")}.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created plot for {country}")
def main():
    """Main function to load data and plot for specified countries."""
    countries = {
        'Guinea': 'ebola_cases_guinea.dat',
        'Liberia': 'ebola_cases_liberia.dat',
        'Sierra Leone': 'ebola_cases_sierra_leone.dat'
        }

    for country, filename in countries.items():
        days, new_cases, progress = load_ebola_data(filename)
        if days is not None:
            print(f"  ✓ Total cases: {progress[-1]:.0f}")
            print(f"  ✓ Peak daily cases: {np.max(new_cases):.0f}")
            print(f"  ✓ Duration: {days[-1]} days")
            
            plot_ebola_data(country, days, new_cases, progress)
        else:
            print(f"Failed to load data for {country}")

if __name__ == "__main__":
    main()
