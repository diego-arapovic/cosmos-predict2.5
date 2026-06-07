
import wandb
api = wandb.Api()
run = api.run("adiego/world2action/d9psqgwk")                  # add entity if needed: "your-entity/world2action"
# run = sorted(runs, key=lambda r: r.created_at)
print("exporting:", run.name, run.id, "| iters:", run.summary.get("iteration"))
run.history(samples=1000000, pandas=True).to_csv("/home/ubuntu/cosmos-predict2.5/w2a_history_d9psqgwk.csv", index=False)
print("wrote /home/ubuntu/cosmos-predict2.5/w2a_history_d9psqgwk.csv")