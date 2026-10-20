import rclpy
from rclpy.node import Node

from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition


class AmclLifecycleSupervisor(Node):

    def __init__(self):
        super().__init__("amcl_lifecycle_supervisor")

        self.get_state_client = self.create_client(
            GetState,
            "/amcl/get_state",
        )

        self.change_state_client = self.create_client(
            ChangeState,
            "/amcl/change_state",
        )

        self.busy = False
        self.last_state = None
        self.request_generation = 0

        # Vérification fréquente pour récupérer rapidement AMCL
        # après un respawn.
        self.timer = self.create_timer(
            1.0,
            self.check_amcl,
        )

        self.get_logger().info(
            "[G1 AMCL SUPERVISOR] Started."
        )

    def check_amcl(self):

        if self.busy:
            return

        if not self.get_state_client.service_is_ready():
            self.get_logger().info(
                "[G1 AMCL SUPERVISOR] "
                "Waiting for /amcl/get_state..."
            )
            return

        self.busy = True
        self.request_generation += 1

        generation = self.request_generation

        self.get_logger().info(
            "[G1 AMCL SUPERVISOR] "
            "Checking AMCL lifecycle state..."
        )

        future = self.get_state_client.call_async(
            GetState.Request()
        )

        future.add_done_callback(
            lambda f: self.state_received(f, generation)
        )

        # Watchdog :
        # si AMCL meurt pendant la requête, on ne doit jamais
        # laisser self.busy bloqué indéfiniment.
        self.create_timer(
            1.5,
            lambda: self.request_timeout(generation),
        )

    def request_timeout(self, generation):

        if generation != self.request_generation:
            return

        if self.busy:
            self.get_logger().warning(
                "[G1 AMCL SUPERVISOR] "
                "get_state timeout -> releasing supervisor lock."
            )

            self.busy = False

    def state_received(self, future, generation):

        if generation != self.request_generation:
            return

        try:
            response = future.result()

        except Exception as exc:
            self.get_logger().warning(
                f"[G1 AMCL SUPERVISOR] "
                f"get_state failed: {exc}"
            )

            self.busy = False
            return

        state = response.current_state.id
        label = response.current_state.label

        self.get_logger().info(
            f"[G1 AMCL SUPERVISOR] "
            f"AMCL state: {label} [{state}]"
        )

        self.last_state = state

        # -------------------------------------------------
        # UNCONFIGURED -> CONFIGURE
        # -------------------------------------------------

        if state == 1:

            self.get_logger().warning(
                "[G1 AMCL SUPERVISOR] "
                "AMCL is UNCONFIGURED -> CONFIGURE"
            )

            self.request_transition(
                Transition.TRANSITION_CONFIGURE,
                generation,
            )

            return

        # -------------------------------------------------
        # INACTIVE -> ACTIVATE
        # -------------------------------------------------

        if state == 2:

            self.get_logger().warning(
                "[G1 AMCL SUPERVISOR] "
                "AMCL is INACTIVE -> ACTIVATE"
            )

            self.request_transition(
                Transition.TRANSITION_ACTIVATE,
                generation,
            )

            return

        # -------------------------------------------------
        # ACTIVE
        # -------------------------------------------------

        if state == 3:

            self.get_logger().info(
                "[G1 AMCL SUPERVISOR] "
                "AMCL is ACTIVE."
            )

        self.busy = False

    def request_transition(self, transition_id, generation):

        if not self.change_state_client.service_is_ready():

            self.get_logger().warning(
                "[G1 AMCL SUPERVISOR] "
                "/amcl/change_state not ready."
            )

            self.busy = False
            return

        request = ChangeState.Request()
        request.transition.id = transition_id

        future = self.change_state_client.call_async(
            request
        )

        future.add_done_callback(
            lambda f: self.transition_done(
                f,
                transition_id,
                generation,
            )
        )

    def transition_done(
        self,
        future,
        transition_id,
        generation,
    ):

        if generation != self.request_generation:
            return

        try:
            response = future.result()

            if response.success:

                self.get_logger().info(
                    f"[G1 AMCL SUPERVISOR] "
                    f"Transition {transition_id} successful."
                )

            else:

                self.get_logger().warning(
                    f"[G1 AMCL SUPERVISOR] "
                    f"Transition {transition_id} failed."
                )

        except Exception as exc:

            self.get_logger().warning(
                f"[G1 AMCL SUPERVISOR] "
                f"transition failed: {exc}"
            )

        self.busy = False


def main(args=None):

    rclpy.init(args=args)

    node = AmclLifecycleSupervisor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()