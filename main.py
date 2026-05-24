from data import create_data
from ab_test import run_ab_test, print_ab_results
from model import train_t_learner
from evaluation import summarize_uplift, print_uplift_summary, plot_qini_curve
from analysis import segment_analysis, print_segment_analysis


def main():
    df = create_data()

    print(df.head())
    print("\nDataset shape:", df.shape)

    ab_results = run_ab_test(df)
    print_ab_results(ab_results)

    result, control_model, treatment_model = train_t_learner(df)

    uplift_summary = summarize_uplift(result)
    print_uplift_summary(uplift_summary)

    segment_summary, best_users = segment_analysis(result)
    print_segment_analysis(segment_summary, best_users)

    plot_qini_curve(result)


if __name__ == "__main__":
    main()