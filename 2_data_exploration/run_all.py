"""
Run all data exploration scripts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_studies import main as run_studies
from explore_sponsors import main as run_sponsors
from explore_browse_conditions import main as run_browse_conditions
from explore_interventions import main as run_interventions
from explore_eligibilities import main as run_eligibilities
from explore_site_footprint import main as run_site_footprint
from explore_designs import main as run_designs
from explore_arm_intervention import main as run_arm_intervention
from explore_design_outcomes import main as run_design_outcomes
from explore_eligibility_criteria_text import main as run_eligibility_criteria_text
from explore_max_planned_followup_days import main as run_max_planned_followup_days

if __name__ == "__main__":
    run_studies()
    run_sponsors()
    run_browse_conditions()
    run_interventions()
    run_eligibilities()
    run_site_footprint()
    run_designs()
    run_design_outcomes()
    run_arm_intervention()
    run_eligibility_criteria_text()
    run_max_planned_followup_days()
